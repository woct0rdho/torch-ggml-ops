using torch::headeronly::ScalarType;
using torch::stable::Tensor;

int64_t packed_block_bytes(int64_t quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
        default:
            STD_TORCH_CHECK(false, "unsupported quant_type: ", quant_type);
    }
}

int64_t packed_block_values(int64_t quant_type) {
    return quant_type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
}

int64_t packed_row_bytes(int64_t quant_type, int64_t in_features) {
    return (in_features / packed_block_values(quant_type)) *
        packed_block_bytes(quant_type);
}

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

struct DenseMMQShape {
    int rows;
    int in_features;
    int out_features;
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
        0,
        static_cast<int32_t>(quant_type),
        static_cast<int>(rows),
        static_cast<int>(out_features),
        static_cast<int>(in_features));
    return {
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
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
        1,
        static_cast<int32_t>(quant_type),
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(out_features));
    return {
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        0,
    };
}

