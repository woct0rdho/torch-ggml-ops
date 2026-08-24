struct GroupedMMQShape {
    int rows;
    int in_features;
    int out_features;
    int num_experts;
    int num_groups;
    int64_t bytes_per_expert;
};

GroupedMMQShape validate_grouped_mmq(
        const Tensor & input,
        const Tensor & packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t out_features) {
    STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(expert_indices.is_cuda(), "expert_indices must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(expert_offsets.is_cuda(), "expert_offsets must be a CUDA/HIP tensor");
    const int32_t device_index = input.get_device_index();
    STD_TORCH_CHECK(
        packed_weight.get_device_index() == device_index &&
        expert_indices.get_device_index() == device_index &&
        expert_offsets.get_device_index() == device_index,
        "all grouped MMQ tensors must be on the same device");
    STD_TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16, "input must have dtype torch.bfloat16");
    STD_TORCH_CHECK(packed_weight.scalar_type() == ScalarType::Byte, "packed_weight must have dtype torch.uint8");
    STD_TORCH_CHECK(expert_indices.scalar_type() == ScalarType::Long, "expert_indices must have dtype torch.int64");
    STD_TORCH_CHECK(expert_offsets.scalar_type() == ScalarType::Int, "expert_offsets must have dtype torch.int32");
    STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(expert_indices.is_contiguous(), "expert_indices must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(expert_offsets.is_contiguous(), "expert_offsets must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(input.dim() == 2, "grouped MMQ input must have shape [rows, in_features]");
    STD_TORCH_CHECK(packed_weight.dim() == 3, "grouped packed_weight must have physical shape [experts, out_features, row_bytes]");
    STD_TORCH_CHECK(expert_indices.dim() == 1, "expert_indices must be one-dimensional");
    STD_TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be one-dimensional");
    STD_TORCH_CHECK(expert_indices.numel() == expert_offsets.numel(), "expert_indices and expert_offsets must have equal lengths");
    STD_TORCH_CHECK(expert_indices.numel() > 0, "grouped MMQ requires at least one active expert");

    const int64_t rows = input.size(0);
    const int64_t in_features = input.size(1);
    const int64_t num_experts = packed_weight.size(0);
    const int64_t num_groups = expert_indices.numel();
    STD_TORCH_CHECK(rows > 0, "zero-row inputs are not supported");
    STD_TORCH_CHECK(in_features > 0 && in_features % QK_K == 0, "input width must be a positive multiple of 256; got ", in_features);
    STD_TORCH_CHECK(out_features > 0, "out_features must be positive; got ", out_features);
    STD_TORCH_CHECK(
        num_experts == 256,
        "exact grouped deployment requires 256 physical experts");
    STD_TORCH_CHECK(
        num_groups > 0 && num_groups <= 256,
        "exact grouped deployment requires between 1 and 256 route entries");
    STD_TORCH_CHECK(num_groups <= num_experts, "active expert count exceeds packed expert count");
    STD_TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "input row count exceeds the kernel limit");
    STD_TORCH_CHECK(in_features <= std::numeric_limits<int>::max(), "in_features exceeds the kernel limit");
    STD_TORCH_CHECK(out_features <= std::numeric_limits<int>::max(), "out_features exceeds the kernel limit");
    STD_TORCH_CHECK(num_experts <= std::numeric_limits<int>::max(), "expert count exceeds the kernel limit");
    STD_TORCH_CHECK(num_groups <= std::numeric_limits<int>::max(), "active expert count exceeds the kernel limit");

    const int64_t row_bytes = packed_row_bytes(quant_type, in_features);
    STD_TORCH_CHECK(packed_weight.size(1) == out_features, "packed_weight physical output dimension does not match out_features");
    STD_TORCH_CHECK(
        packed_weight.size(2) == row_bytes,
        "packed_weight has ",
        packed_weight.size(2),
        " bytes per row, expected ",
        row_bytes,
        " for in_features=",
        in_features,
        " quant_type=",
        quant_type);
    const int64_t bytes_per_expert = out_features * row_bytes;
    STD_TORCH_CHECK(
        packed_weight.numel() == num_experts * bytes_per_expert,
        "packed_weight byte count is inconsistent with its grouped physical shape");

    const auto input_address = reinterpret_cast<uintptr_t>(input.const_data_ptr());
    const auto packed_address = reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr());
    const auto indices_address =
        reinterpret_cast<uintptr_t>(expert_indices.const_data_ptr());
    const auto offsets_address =
        reinterpret_cast<uintptr_t>(expert_offsets.const_data_ptr());
    STD_TORCH_CHECK(input_address % 16 == 0, "input data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(packed_address % 16 == 0, "packed_weight data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(
        indices_address % alignof(int64_t) == 0,
        "expert_indices data pointer has invalid alignment");
    STD_TORCH_CHECK(
        offsets_address % alignof(int32_t) == 0,
        "expert_offsets data pointer has invalid alignment");

    return {
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        static_cast<int>(num_experts),
        static_cast<int>(num_groups),
        bytes_per_expert,
    };
}

GroupedMMQShape validate_grouped_mmq_grad_input(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t in_features) {
    STD_TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(expert_indices.is_cuda(), "expert_indices must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(expert_offsets.is_cuda(), "expert_offsets must be a CUDA/HIP tensor");
    const int32_t device_index = grad_output.get_device_index();
    STD_TORCH_CHECK(
        packed_weight.get_device_index() == device_index &&
        expert_indices.get_device_index() == device_index &&
        expert_offsets.get_device_index() == device_index,
        "all grouped MMQ backward tensors must be on the same device");
    STD_TORCH_CHECK(
        grad_output.scalar_type() == ScalarType::BFloat16,
        "grad_output must have dtype torch.bfloat16");
    STD_TORCH_CHECK(
        packed_weight.scalar_type() == ScalarType::Byte,
        "packed_weight must have dtype torch.uint8");
    STD_TORCH_CHECK(
        expert_indices.scalar_type() == ScalarType::Long,
        "expert_indices must have dtype torch.int64");
    STD_TORCH_CHECK(
        expert_offsets.scalar_type() == ScalarType::Int,
        "expert_offsets must have dtype torch.int32");
    STD_TORCH_CHECK(
        grad_output.is_contiguous(),
        "grad_output must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(
        packed_weight.is_contiguous(),
        "packed_weight must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(
        expert_indices.is_contiguous(),
        "expert_indices must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(
        expert_offsets.is_contiguous(),
        "expert_offsets must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(
        grad_output.dim() == 2,
        "grouped MMQ grad_output must have shape [rows, out_features]");
    STD_TORCH_CHECK(
        packed_weight.dim() == 3,
        "grouped packed_weight must have physical shape [experts, out_features, row_bytes]");
    STD_TORCH_CHECK(expert_indices.dim() == 1, "expert_indices must be one-dimensional");
    STD_TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be one-dimensional");
    STD_TORCH_CHECK(
        expert_indices.numel() == expert_offsets.numel(),
        "expert_indices and expert_offsets must have equal lengths");
    STD_TORCH_CHECK(
        expert_indices.numel() > 0,
        "grouped MMQ backward requires at least one active expert");

    const int64_t rows = grad_output.size(0);
    const int64_t out_features = grad_output.size(1);
    const int64_t num_experts = packed_weight.size(0);
    const int64_t num_groups = expert_indices.numel();
    STD_TORCH_CHECK(rows > 0, "zero-row grad_output tensors are not supported");
    STD_TORCH_CHECK(out_features > 0, "grad_output final dimension must be positive");
    STD_TORCH_CHECK(
        in_features > 0 && in_features % QK_K == 0,
        "in_features must be a positive multiple of 256; got ",
        in_features);
    STD_TORCH_CHECK(
        num_experts == 256,
        "exact grouped deployment requires 256 physical experts");
    STD_TORCH_CHECK(
        num_groups > 0 && num_groups <= 256,
        "exact grouped deployment requires between 1 and 256 route entries");
    STD_TORCH_CHECK(num_groups <= num_experts, "active expert count exceeds packed expert count");
    STD_TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "grad_output row count exceeds the kernel limit");
    STD_TORCH_CHECK(in_features <= std::numeric_limits<int>::max(), "in_features exceeds the kernel limit");
    STD_TORCH_CHECK(out_features <= std::numeric_limits<int>::max(), "out_features exceeds the kernel limit");
    STD_TORCH_CHECK(num_experts <= std::numeric_limits<int>::max(), "expert count exceeds the kernel limit");
    STD_TORCH_CHECK(num_groups <= std::numeric_limits<int>::max(), "active expert count exceeds the kernel limit");

    const int64_t row_bytes = packed_row_bytes(quant_type, in_features);
    STD_TORCH_CHECK(
        packed_weight.size(1) == out_features,
        "packed_weight physical output dimension does not match grad_output");
    STD_TORCH_CHECK(
        packed_weight.size(2) == row_bytes,
        "packed_weight has ",
        packed_weight.size(2),
        " bytes per row, expected ",
        row_bytes,
        " for in_features=",
        in_features,
        " quant_type=",
        quant_type);
    const int64_t bytes_per_expert = out_features * row_bytes;
    STD_TORCH_CHECK(
        packed_weight.numel() == num_experts * bytes_per_expert,
        "packed_weight byte count is inconsistent with its grouped physical shape");

    const auto grad_address = reinterpret_cast<uintptr_t>(grad_output.const_data_ptr());
    const auto packed_address = reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr());
    const auto indices_address =
        reinterpret_cast<uintptr_t>(expert_indices.const_data_ptr());
    const auto offsets_address =
        reinterpret_cast<uintptr_t>(expert_offsets.const_data_ptr());
    STD_TORCH_CHECK(grad_address % 16 == 0, "grad_output data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(packed_address % 16 == 0, "packed_weight data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(
        indices_address % alignof(int64_t) == 0,
        "expert_indices data pointer has invalid alignment");
    STD_TORCH_CHECK(
        offsets_address % alignof(int32_t) == 0,
        "expert_offsets data pointer has invalid alignment");

    return {
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        static_cast<int>(num_experts),
        static_cast<int>(num_groups),
        bytes_per_expert,
    };
}

