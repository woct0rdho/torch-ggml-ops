void grouped_mmq_pair_launch_cuda(
        const Tensor & input,
        const Tensor & first_packed_weight,
        const Tensor & second_packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t out_features,
        Tensor first_output,
        Tensor second_output,
        Tensor workspace,
        Tensor task_count,
        Tensor task_experts,
        Tensor task_row_starts,
        Tensor task_row_ends) {
    const GroupedMMQShape shape = validate_grouped_mmq(
        input,
        first_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        out_features);
    const GroupedMMQShape second_shape = validate_grouped_mmq(
        input,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        out_features);
    STD_TORCH_CHECK(
        shape.bytes_per_expert == second_shape.bytes_per_expert,
        "paired packed weights have different physical contracts");
    torch_ggml_ops::mmq_bundle::require_exact_deployment(
        3, static_cast<int32_t>(quant_type), shape.rows,
        shape.out_features, shape.in_features);
    const int row_task_rows =
        torch_ggml_ops::mmq_bundle::exact_deployment_row_task_rows(
            3, static_cast<int32_t>(quant_type), shape.rows,
            shape.out_features, shape.in_features);
    const int max_tasks = row_task_rows == 0
        ? 0
        : (shape.rows + row_task_rows - 1) / row_task_rows + shape.num_groups;
    if (max_tasks != 0) {
        STD_TORCH_CHECK(
            max_tasks <=
                torch_ggml_ops::mmq_bundle::exact_deployment_row_task_capacity(
                    3, static_cast<int32_t>(quant_type), shape.rows,
                    shape.out_features, shape.in_features),
            "paired grouped row-task capacity exceeds the deployment bound");
    }
    const int64_t workspace_bytes = static_cast<int64_t>(shape.rows) *
        (shape.in_features / kQuantWorkspaceBlockValues) *
        kQuantWorkspaceBlockBytes;
    for (Tensor * output : {&first_output, &second_output}) {
        validate_explicit_buffer(
            *output,
            input,
            ScalarType::BFloat16,
            static_cast<int64_t>(shape.rows) * shape.out_features,
            "paired output");
        STD_TORCH_CHECK(
            output->dim() == 2 && output->size(0) == shape.rows &&
                output->size(1) == shape.out_features,
            "paired output has an invalid shape");
    }
    validate_explicit_vector(
        workspace, input, ScalarType::Byte, workspace_bytes, "workspace");
    validate_explicit_vector(
        task_count, input, ScalarType::Int, max_tasks == 0 ? 0 : 1, "task_count");
    validate_explicit_vector(
        task_experts, input, ScalarType::Int, max_tasks, "task_experts");
    validate_explicit_vector(
        task_row_starts, input, ScalarType::Int, max_tasks, "task_row_starts");
    validate_explicit_vector(
        task_row_ends, input, ScalarType::Int, max_tasks, "task_row_ends");
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
    if (row_task_rows != 0) {
        torch_ggml_ops::mmq_bundle::launch_grouped_row_task_setup(
            static_cast<const int64_t *>(expert_indices.const_data_ptr()),
            static_cast<const int32_t *>(expert_offsets.const_data_ptr()),
            static_cast<int32_t *>(task_count.mutable_data_ptr()),
            static_cast<int32_t *>(task_experts.mutable_data_ptr()),
            static_cast<int32_t *>(task_row_starts.mutable_data_ptr()),
            static_cast<int32_t *>(task_row_ends.mutable_data_ptr()),
            shape.num_experts,
            shape.num_groups,
            shape.rows,
            row_task_rows,
            stream);
        torch_ggml_ops::mmq_bundle::launch_grouped_pair_forward_row_tasks(
            static_cast<int32_t>(quant_type),
            static_cast<const char *>(first_packed_weight.const_data_ptr()),
            static_cast<const char *>(second_packed_weight.const_data_ptr()),
            static_cast<const int *>(workspace.const_data_ptr()),
            static_cast<__hip_bfloat16 *>(first_output.mutable_data_ptr()),
            static_cast<__hip_bfloat16 *>(second_output.mutable_data_ptr()),
            static_cast<const int32_t *>(task_count.const_data_ptr()),
            static_cast<const int32_t *>(task_experts.const_data_ptr()),
            static_cast<const int32_t *>(task_row_starts.const_data_ptr()),
            static_cast<const int32_t *>(task_row_ends.const_data_ptr()),
            max_tasks,
            shape.num_experts,
            shape.rows,
            shape.in_features,
            shape.out_features,
            shape.bytes_per_expert,
            stream);
        return;
    }
    torch_ggml_ops::mmq_bundle::launch_grouped_pair_forward(
        static_cast<int32_t>(quant_type),
        static_cast<const char *>(first_packed_weight.const_data_ptr()),
        static_cast<const char *>(second_packed_weight.const_data_ptr()),
        static_cast<const int *>(workspace.const_data_ptr()),
        static_cast<__hip_bfloat16 *>(first_output.mutable_data_ptr()),
        static_cast<__hip_bfloat16 *>(second_output.mutable_data_ptr()),
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

void grouped_mmq_pair_grad_input_launch_cuda(
        const Tensor & first_grad_output,
        const Tensor & second_grad_output,
        const Tensor & first_packed_weight,
        const Tensor & second_packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t in_features,
        Tensor grad_input) {
    const GroupedMMQShape shape = validate_grouped_mmq_grad_input(
        first_grad_output,
        first_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features);
    const GroupedMMQShape second_shape = validate_grouped_mmq_grad_input(
        second_grad_output,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features);
    STD_TORCH_CHECK(
        shape.rows == second_shape.rows &&
            shape.out_features == second_shape.out_features &&
            shape.bytes_per_expert == second_shape.bytes_per_expert,
        "paired grouped gradients have different exact problems");
    torch_ggml_ops::mmq_bundle::require_exact_deployment(
        5, static_cast<int32_t>(quant_type), shape.rows,
        shape.in_features, shape.out_features);
    validate_explicit_buffer(
        grad_input,
        first_grad_output,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.rows) * shape.in_features,
        "grad_input");
    STD_TORCH_CHECK(
        grad_input.dim() == 2 && grad_input.size(0) == shape.rows &&
            grad_input.size(1) == shape.in_features,
        "grad_input has an invalid paired grouped shape");
    torch::stable::accelerator::DeviceGuard guard(
        first_grad_output.get_device_index());
    torch_ggml_ops::mmq_bundle::launch_grouped_pair_backward(
        static_cast<int32_t>(quant_type),
        static_cast<const __hip_bfloat16 *>(first_grad_output.const_data_ptr()),
        static_cast<const __hip_bfloat16 *>(second_grad_output.const_data_ptr()),
        static_cast<const char *>(first_packed_weight.const_data_ptr()),
        static_cast<const char *>(second_packed_weight.const_data_ptr()),
        static_cast<__hip_bfloat16 *>(grad_input.mutable_data_ptr()),
        static_cast<const int64_t *>(expert_indices.const_data_ptr()),
        static_cast<const int32_t *>(expert_offsets.const_data_ptr()),
        shape.num_experts,
        shape.num_groups,
        shape.rows,
        shape.out_features,
        shape.in_features,
        shape.bytes_per_expert,
        current_stream(first_grad_output));
}

