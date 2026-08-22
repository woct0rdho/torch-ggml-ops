struct FixedMMQShape {
    int tokens;
    int total_rows;
    int out_features;
    int64_t bytes_per_group;
};

FixedMMQShape validate_fixed_mmq(
        const Tensor & tensor,
        const Tensor & packed_weight,
        bool backward) {
    constexpr int groups = 8;
    constexpr int in_features = 4096;
    constexpr int row_bytes = (in_features / QK8_0) * sizeof(block_q8_0);
    STD_TORCH_CHECK(tensor.is_cuda(), "fixed grouped input must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(
        tensor.get_device_index() == packed_weight.get_device_index(),
        "fixed grouped tensors must be on the same device");
    STD_TORCH_CHECK(tensor.scalar_type() == ScalarType::BFloat16, "fixed grouped input must be BF16");
    STD_TORCH_CHECK(packed_weight.scalar_type() == ScalarType::Byte, "packed_weight must be uint8");
    STD_TORCH_CHECK(tensor.is_contiguous(), "fixed grouped input must be contiguous");
    STD_TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
    STD_TORCH_CHECK(tensor.storage_offset() == 0, "fixed grouped input must have zero storage offset");
    STD_TORCH_CHECK(packed_weight.storage_offset() == 0, "packed_weight must have zero storage offset");
    STD_TORCH_CHECK(tensor.dim() >= 2, "fixed grouped input must have at least two dimensions");
    STD_TORCH_CHECK(
        tensor.size(tensor.dim() - 2) == groups,
        "fixed grouped input must have eight groups");
    STD_TORCH_CHECK(
        packed_weight.dim() == 3 && packed_weight.size(0) == groups &&
            packed_weight.size(2) == row_bytes,
        "fixed grouped packed_weight has an invalid physical shape");
    const int64_t out_features = packed_weight.size(1);
    const int64_t expected_width = backward ? out_features : in_features;
    STD_TORCH_CHECK(
        tensor.size(tensor.dim() - 1) == expected_width,
        "fixed grouped input width does not match the deployment contract");
    const int64_t total_rows = tensor.numel() / expected_width;
    const int64_t tokens = total_rows / groups;
    STD_TORCH_CHECK(tokens > 0 && out_features > 0, "fixed grouped dimensions must be positive");
    STD_TORCH_CHECK(
        tokens <= std::numeric_limits<int>::max() &&
            total_rows <= std::numeric_limits<int>::max() &&
            out_features <= std::numeric_limits<int>::max(),
        "fixed grouped dimensions exceed the kernel limit");
    const int64_t bytes_per_group = out_features * row_bytes;
    STD_TORCH_CHECK(
        packed_weight.numel() == groups * bytes_per_group,
        "fixed grouped packed_weight byte count is inconsistent");
    STD_TORCH_CHECK(
        reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr()) % 16 == 0,
        "fixed grouped tensor pointers must be 16-byte aligned");
    torch_ggml_ops::mmq_bundle::require_exact_deployment(
        backward ? 7 : 6,
        GGML_TYPE_Q8_0,
        static_cast<int>(tokens),
        backward ? in_features : static_cast<int>(out_features),
        backward ? static_cast<int>(out_features) : in_features);
    return {
        static_cast<int>(tokens),
        static_cast<int>(total_rows),
        static_cast<int>(out_features),
        bytes_per_group,
    };
}

