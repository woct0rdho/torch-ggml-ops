// torch.utils.cpp_extension disables HIP half operators/conversions by default.
// llama.cpp MMQ deliberately uses the native half/half2 arithmetic surface.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include "mmq_core.cuh"
#include "ck/grouped_mmq_backward.cuh"
#include "ck/mmq_backward.cuh"

#include <hip/hip_runtime.h>
#include <Python.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

int64_t packed_block_bytes(int64_t quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
        default:
            STD_TORCH_CHECK(false, "unsupported quant_type: ", quant_type);
    }
}

template <typename Function>
void dispatch_quant_type(int64_t quant_type, Function && function) {
    switch (quant_type) {
        case GGML_TYPE_Q3_K:
            function(std::integral_constant<ggml_type, GGML_TYPE_Q3_K>{});
            break;
        case GGML_TYPE_Q4_K:
            function(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{});
            break;
        case GGML_TYPE_Q5_K:
            function(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{});
            break;
        case GGML_TYPE_Q6_K:
            function(std::integral_constant<ggml_type, GGML_TYPE_Q6_K>{});
            break;
        case GGML_TYPE_IQ2_S:
            function(std::integral_constant<ggml_type, GGML_TYPE_IQ2_S>{});
            break;
        default:
            STD_TORCH_CHECK(false, "unsupported quant_type: ", quant_type);
    }
}

void check_hip(hipError_t status, const char * operation) {
    STD_TORCH_CHECK(status == hipSuccess, operation, " failed: ", hipGetErrorString(status));
}

int sram_stride_host(int64_t quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ2_S:
            return mmq_sram_stride(GGML_TYPE_Q3_K);
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return mmq_sram_stride(GGML_TYPE_Q4_K);
        case GGML_TYPE_Q6_K:
            return mmq_sram_stride(GGML_TYPE_Q6_K);
        default:
            return -1;
    }
}

template <ggml_type type, int J>
void launch_dense_mmq_multiply(
        const char * packed,
        __hip_bfloat16 * output,
        const block_q8_1_mmq * workspace,
        int rows,
        int rows_padded,
        int in_features,
        int out_features,
        hipStream_t stream) {
    const dim3 mmq_grid((out_features + MMQ_I - 1) / MMQ_I, rows_padded / J, 1);
    const dim3 mmq_block(WARP_SIZE, MMQ_NWARPS, 1);
    const int shared_ints = J + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS)
        + MMQ_I * sram_stride_host(type);
    dense_mmq_bf16_kernel<type, J>
        <<<mmq_grid, mmq_block, shared_ints * sizeof(int), stream>>>(
            packed,
            reinterpret_cast<const int *>(workspace),
            output,
            out_features,
            rows,
            rows_padded,
            in_features / QK_K);
}

template <ggml_type type>
void launch_dense_mmq(
        const __hip_bfloat16 * input,
        const char * packed,
        __hip_bfloat16 * output,
        block_q8_1_mmq * workspace,
        int rows,
        int rows_padded,
        int in_features,
        int out_features,
        hipStream_t stream) {
    const dim3 quant_grid(rows, 1, 1);
    const dim3 quant_block(512, 1, 1);
    quantize_bf16_mmq_q8_1<type><<<quant_grid, quant_block, 0, stream>>>(
        input, workspace, rows, rows_padded, in_features);
    check_hip(hipGetLastError(), "quantize_bf16_mmq_q8_1 launch");

    if constexpr (type == GGML_TYPE_Q6_K) {
        if (rows_padded == MMQ_J_SMALL) {
            launch_dense_mmq_multiply<type, MMQ_J_SMALL>(
                packed, output, workspace, rows, rows_padded, in_features, out_features, stream);
        } else {
            launch_dense_mmq_multiply<type, MMQ_J>(
                packed, output, workspace, rows, rows_padded, in_features, out_features, stream);
        }
    } else {
        launch_dense_mmq_multiply<type, MMQ_J>(
            packed, output, workspace, rows, rows_padded, in_features, out_features, stream);
    }
    check_hip(hipGetLastError(), "dense_mmq_bf16_kernel launch");
}

template <ggml_type type>
void launch_grouped_quantize(
        const __hip_bfloat16 * input,
        block_q8_1_mmq * workspace,
        int rows,
        int in_features,
        hipStream_t stream) {
    const dim3 quant_grid(rows, 1, 1);
    const dim3 quant_block(512, 1, 1);
    quantize_bf16_mmq_q8_1<type><<<quant_grid, quant_block, 0, stream>>>(
        input, workspace, rows, rows, in_features);
    check_hip(hipGetLastError(), "grouped quantize_bf16_mmq_q8_1 launch");
}

template <ggml_type type>
void launch_grouped_projection(
        const char * packed,
        const int * activations,
        __hip_bfloat16 * output,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int in_features,
        int out_features,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 mmq_grid((out_features + MMQ_I - 1) / MMQ_I, num_groups, 1);
    const dim3 mmq_block(WARP_SIZE, MMQ_NWARPS, 1);
    const int shared_ints = MMQ_J + GGML_PAD(MMQ_J * MMQ_TILE_Y_K, MMQ_NTHREADS)
        + MMQ_I * sram_stride_host(type);
    grouped_mmq_bf16_kernel<type><<<mmq_grid, mmq_block, shared_ints * sizeof(int), stream>>>(
        packed,
        activations,
        output,
        expert_indices,
        expert_offsets,
        num_experts,
        out_features,
        rows,
        in_features / QK_K,
        bytes_per_expert);
    check_hip(hipGetLastError(), "grouped_mmq_bf16_kernel launch");
}

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
    STD_TORCH_CHECK(input.storage_offset() == 0, "input must have zero storage offset");
    STD_TORCH_CHECK(packed_weight.storage_offset() == 0, "packed_weight must have zero storage offset");
    STD_TORCH_CHECK(expert_indices.storage_offset() == 0, "expert_indices must have zero storage offset");
    STD_TORCH_CHECK(expert_offsets.storage_offset() == 0, "expert_offsets must have zero storage offset");
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
    STD_TORCH_CHECK(num_experts > 0, "packed_weight must contain at least one expert");
    STD_TORCH_CHECK(num_groups <= num_experts, "active expert count exceeds packed expert count");
    STD_TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "input row count exceeds the kernel limit");
    STD_TORCH_CHECK(in_features <= std::numeric_limits<int>::max(), "in_features exceeds the kernel limit");
    STD_TORCH_CHECK(out_features <= std::numeric_limits<int>::max(), "out_features exceeds the kernel limit");
    STD_TORCH_CHECK(num_experts <= std::numeric_limits<int>::max(), "expert count exceeds the kernel limit");
    STD_TORCH_CHECK(num_groups <= std::numeric_limits<int>::max(), "active expert count exceeds the kernel limit");

    const int64_t row_bytes = (in_features / QK_K) * packed_block_bytes(quant_type);
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
    STD_TORCH_CHECK(input_address % 16 == 0, "input data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(packed_address % 16 == 0, "packed_weight data pointer must be 16-byte aligned");

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
    STD_TORCH_CHECK(grad_output.storage_offset() == 0, "grad_output must have zero storage offset");
    STD_TORCH_CHECK(packed_weight.storage_offset() == 0, "packed_weight must have zero storage offset");
    STD_TORCH_CHECK(expert_indices.storage_offset() == 0, "expert_indices must have zero storage offset");
    STD_TORCH_CHECK(expert_offsets.storage_offset() == 0, "expert_offsets must have zero storage offset");
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
    STD_TORCH_CHECK(num_experts > 0, "packed_weight must contain at least one expert");
    STD_TORCH_CHECK(num_groups <= num_experts, "active expert count exceeds packed expert count");
    STD_TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "grad_output row count exceeds the kernel limit");
    STD_TORCH_CHECK(in_features <= std::numeric_limits<int>::max(), "in_features exceeds the kernel limit");
    STD_TORCH_CHECK(out_features <= std::numeric_limits<int>::max(), "out_features exceeds the kernel limit");
    STD_TORCH_CHECK(num_experts <= std::numeric_limits<int>::max(), "expert count exceeds the kernel limit");
    STD_TORCH_CHECK(num_groups <= std::numeric_limits<int>::max(), "active expert count exceeds the kernel limit");

    const int64_t row_bytes =
        (in_features / QK_K) * packed_block_bytes(quant_type);
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
    STD_TORCH_CHECK(grad_address % 16 == 0, "grad_output data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(packed_address % 16 == 0, "packed_weight data pointer must be 16-byte aligned");

    return {
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        static_cast<int>(num_experts),
        static_cast<int>(num_groups),
        bytes_per_expert,
    };
}

Tensor new_grouped_output(const Tensor & input, const GroupedMMQShape & shape) {
    std::array<int64_t, 2> output_sizes{shape.rows, shape.out_features};
    return torch::stable::new_empty(
        input,
        torch::headeronly::IntHeaderOnlyArrayRef(output_sizes.data(), output_sizes.size()),
        ScalarType::BFloat16);
}

Tensor new_grouped_workspace(const Tensor & input, const GroupedMMQShape & shape) {
    const int64_t workspace_bytes =
        static_cast<int64_t>(shape.rows) * (shape.in_features / (4 * QK8_1)) * sizeof(block_q8_1_mmq);
    std::array<int64_t, 1> workspace_size{workspace_bytes};
    return torch::stable::new_empty(
        input,
        torch::headeronly::IntHeaderOnlyArrayRef(workspace_size.data(), workspace_size.size()),
        ScalarType::Byte);
}

Tensor mmq_cuda(
        const Tensor & input,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t out_features) {
    STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(input.get_device_index() == packed_weight.get_device_index(), "input and packed_weight must be on the same device");
    STD_TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16, "input must have dtype torch.bfloat16");
    STD_TORCH_CHECK(packed_weight.scalar_type() == ScalarType::Byte, "packed_weight must have dtype torch.uint8");
    STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    STD_TORCH_CHECK(input.storage_offset() == 0, "input must have zero storage offset");
    STD_TORCH_CHECK(packed_weight.storage_offset() == 0, "packed_weight must have zero storage offset");

    const int64_t in_features = input.size(input.dim() - 1);
    STD_TORCH_CHECK(in_features > 0 && in_features % QK_K == 0, "input width must be a positive multiple of 256; got ", in_features);
    STD_TORCH_CHECK(out_features > 0, "out_features must be positive; got ", out_features);
    STD_TORCH_CHECK(input.numel() % in_features == 0, "input numel is inconsistent with its final dimension");
    const int64_t rows = input.numel() / in_features;
    STD_TORCH_CHECK(rows > 0, "zero-row inputs are not supported");
    STD_TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "flattened input row count exceeds the kernel limit");
    STD_TORCH_CHECK(in_features <= std::numeric_limits<int>::max(), "in_features exceeds the kernel limit");
    STD_TORCH_CHECK(out_features <= std::numeric_limits<int>::max(), "out_features exceeds the kernel limit");

    const int64_t block_bytes = packed_block_bytes(quant_type);
    const int64_t expected_packed_bytes = out_features * (in_features / QK_K) * block_bytes;
    STD_TORCH_CHECK(
        packed_weight.numel() == expected_packed_bytes,
        "packed_weight has ",
        packed_weight.numel(),
        " bytes, expected ",
        expected_packed_bytes,
        " for [",
        out_features,
        ", ",
        in_features,
        "] quant_type=",
        quant_type);

    const auto input_address = reinterpret_cast<uintptr_t>(input.const_data_ptr());
    const auto packed_address = reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr());
    STD_TORCH_CHECK(input_address % 16 == 0, "input data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(packed_address % 16 == 0, "packed_weight data pointer must be 16-byte aligned");

    const int32_t device_index = input.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(device_index);

    std::vector<int64_t> output_sizes(input.sizes().begin(), input.sizes().end());
    output_sizes.back() = out_features;
    Tensor output = torch::stable::new_empty(
        input,
        torch::headeronly::IntHeaderOnlyArrayRef(output_sizes.data(), output_sizes.size()),
        ScalarType::BFloat16);

    const int64_t row_tile =
        quant_type == GGML_TYPE_Q6_K && rows <= MMQ_J_SMALL ? MMQ_J_SMALL : MMQ_J;
    const int64_t rows_padded = ((rows + row_tile - 1) / row_tile) * row_tile;
    const int64_t workspace_bytes =
        rows_padded * (in_features / (4 * QK8_1)) * sizeof(block_q8_1_mmq);
    std::array<int64_t, 1> workspace_size{workspace_bytes};
    Tensor workspace_tensor = torch::stable::new_empty(
        input,
        torch::headeronly::IntHeaderOnlyArrayRef(workspace_size.data(), workspace_size.size()),
        ScalarType::Byte);

    void * stream_pointer = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_pointer));
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);

    const auto * input_pointer = static_cast<const __hip_bfloat16 *>(input.const_data_ptr());
    const auto * packed_pointer = static_cast<const char *>(packed_weight.const_data_ptr());
    auto * output_pointer = static_cast<__hip_bfloat16 *>(output.mutable_data_ptr());
    auto * workspace_pointer = static_cast<block_q8_1_mmq *>(workspace_tensor.mutable_data_ptr());

    dispatch_quant_type(quant_type, [&](auto type_tag) {
        constexpr ggml_type type = decltype(type_tag)::value;
        launch_dense_mmq<type>(
            input_pointer,
            packed_pointer,
            output_pointer,
            workspace_pointer,
            static_cast<int>(rows),
            static_cast<int>(rows_padded),
            static_cast<int>(in_features),
            static_cast<int>(out_features),
            stream);
    });

    return output;
}

Tensor mmq_grad_input_cuda(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t in_features) {
    STD_TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(
        grad_output.get_device_index() == packed_weight.get_device_index(),
        "grad_output and packed_weight must be on the same device");
    STD_TORCH_CHECK(
        grad_output.scalar_type() == ScalarType::BFloat16,
        "grad_output must have dtype torch.bfloat16");
    STD_TORCH_CHECK(
        packed_weight.scalar_type() == ScalarType::Byte,
        "packed_weight must have dtype torch.uint8");
    STD_TORCH_CHECK(
        grad_output.is_contiguous(),
        "grad_output must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(
        packed_weight.is_contiguous(),
        "packed_weight must be contiguous; torch_ggml_ops will not insert a hidden copy");
    STD_TORCH_CHECK(grad_output.dim() >= 1, "grad_output must have at least one dimension");
    STD_TORCH_CHECK(grad_output.storage_offset() == 0, "grad_output must have zero storage offset");
    STD_TORCH_CHECK(packed_weight.storage_offset() == 0, "packed_weight must have zero storage offset");
    STD_TORCH_CHECK(
        in_features > 0 && in_features % QK_K == 0,
        "in_features must be a positive multiple of 256; got ",
        in_features);

    const int64_t out_features = grad_output.size(grad_output.dim() - 1);
    STD_TORCH_CHECK(out_features > 0, "grad_output final dimension must be positive");
    STD_TORCH_CHECK(
        grad_output.numel() % out_features == 0,
        "grad_output numel is inconsistent with its final dimension");
    const int64_t rows = grad_output.numel() / out_features;
    STD_TORCH_CHECK(rows > 0, "zero-row grad_output tensors are not supported");
    STD_TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "flattened grad_output row count exceeds the kernel limit");
    STD_TORCH_CHECK(in_features <= std::numeric_limits<int>::max(), "in_features exceeds the kernel limit");
    STD_TORCH_CHECK(out_features <= std::numeric_limits<int>::max(), "out_features exceeds the kernel limit");

    const int64_t block_bytes = packed_block_bytes(quant_type);
    const int64_t expected_packed_bytes =
        out_features * (in_features / QK_K) * block_bytes;
    STD_TORCH_CHECK(
        packed_weight.numel() == expected_packed_bytes,
        "packed_weight has ",
        packed_weight.numel(),
        " bytes, expected ",
        expected_packed_bytes,
        " for [",
        out_features,
        ", ",
        in_features,
        "] quant_type=",
        quant_type);

    const auto grad_address = reinterpret_cast<uintptr_t>(grad_output.const_data_ptr());
    const auto packed_address = reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr());
    STD_TORCH_CHECK(grad_address % 16 == 0, "grad_output data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(packed_address % 16 == 0, "packed_weight data pointer must be 16-byte aligned");

    const int32_t device_index = grad_output.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(device_index);
    std::vector<int64_t> output_sizes(grad_output.sizes().begin(), grad_output.sizes().end());
    output_sizes.back() = in_features;
    Tensor grad_input = torch::stable::new_empty(
        grad_output,
        torch::headeronly::IntHeaderOnlyArrayRef(output_sizes.data(), output_sizes.size()),
        ScalarType::BFloat16);

    void * stream_pointer = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_pointer));
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);

    const auto * grad_pointer =
        static_cast<const __hip_bfloat16 *>(grad_output.const_data_ptr());
    const auto * packed_pointer = static_cast<const char *>(packed_weight.const_data_ptr());
    auto * input_pointer = static_cast<__hip_bfloat16 *>(grad_input.mutable_data_ptr());

    dispatch_quant_type(quant_type, [&](auto type_tag) {
        constexpr ggml_type type = decltype(type_tag)::value;
        torch_ggml_ops::ck::launch_dense_mmq_grad_input<type>(
            grad_pointer,
            packed_pointer,
            input_pointer,
            static_cast<int>(rows),
            static_cast<int>(out_features),
            static_cast<int>(in_features),
            stream);
    });
    check_hip(hipGetLastError(), "dense_mmq_grad_input_kernel launch");

    return grad_input;
}

Tensor grouped_mmq_grad_input_cuda(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t in_features) {
    const GroupedMMQShape shape = validate_grouped_mmq_grad_input(
        grad_output,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features);
    const int32_t device_index = grad_output.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(device_index);
    std::array<int64_t, 2> grad_input_sizes{shape.rows, shape.in_features};
    Tensor grad_input = torch::stable::new_empty(
        grad_output,
        torch::headeronly::IntHeaderOnlyArrayRef(
            grad_input_sizes.data(), grad_input_sizes.size()),
        ScalarType::BFloat16);

    void * stream_pointer = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_pointer));
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);

    const auto * grad_pointer =
        static_cast<const __hip_bfloat16 *>(grad_output.const_data_ptr());
    const auto * packed_pointer =
        static_cast<const char *>(packed_weight.const_data_ptr());
    const auto * expert_pointer =
        static_cast<const int64_t *>(expert_indices.const_data_ptr());
    const auto * offsets_pointer =
        static_cast<const int32_t *>(expert_offsets.const_data_ptr());
    auto * input_pointer =
        static_cast<__hip_bfloat16 *>(grad_input.mutable_data_ptr());

    dispatch_quant_type(quant_type, [&](auto type_tag) {
        constexpr ggml_type type = decltype(type_tag)::value;
        torch_ggml_ops::ck::launch_grouped_mmq_grad_input<type>(
            grad_pointer,
            packed_pointer,
            input_pointer,
            expert_pointer,
            offsets_pointer,
            shape.num_experts,
            shape.num_groups,
            shape.rows,
            shape.out_features,
            shape.in_features,
            shape.bytes_per_expert,
            stream);
    });
    check_hip(hipGetLastError(), "grouped_mmq_grad_input_kernel launch");

    return grad_input;
}

Tensor grouped_mmq_pair_grad_input_cuda(
        const Tensor & first_grad_output,
        const Tensor & second_grad_output,
        const Tensor & first_packed_weight,
        const Tensor & second_packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t in_features) {
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
        second_shape.rows == shape.rows &&
        second_shape.out_features == shape.out_features &&
        second_shape.num_experts == shape.num_experts &&
        second_shape.num_groups == shape.num_groups &&
        second_shape.bytes_per_expert == shape.bytes_per_expert,
        "paired grouped MMQ backward tensors must have identical physical geometry");

    const int32_t device_index = first_grad_output.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(device_index);
    std::array<int64_t, 2> grad_input_sizes{shape.rows, shape.in_features};
    Tensor grad_input = torch::stable::new_empty(
        first_grad_output,
        torch::headeronly::IntHeaderOnlyArrayRef(
            grad_input_sizes.data(), grad_input_sizes.size()),
        ScalarType::BFloat16);

    void * stream_pointer = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_pointer));
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);

    const auto * first_grad_pointer =
        static_cast<const __hip_bfloat16 *>(first_grad_output.const_data_ptr());
    const auto * second_grad_pointer =
        static_cast<const __hip_bfloat16 *>(second_grad_output.const_data_ptr());
    const auto * first_packed_pointer =
        static_cast<const char *>(first_packed_weight.const_data_ptr());
    const auto * second_packed_pointer =
        static_cast<const char *>(second_packed_weight.const_data_ptr());
    const auto * expert_pointer =
        static_cast<const int64_t *>(expert_indices.const_data_ptr());
    const auto * offsets_pointer =
        static_cast<const int32_t *>(expert_offsets.const_data_ptr());
    auto * input_pointer =
        static_cast<__hip_bfloat16 *>(grad_input.mutable_data_ptr());

    dispatch_quant_type(quant_type, [&](auto type_tag) {
        constexpr ggml_type type = decltype(type_tag)::value;
        torch_ggml_ops::ck::launch_grouped_mmq_pair_grad_input<type>(
            first_grad_pointer,
            second_grad_pointer,
            first_packed_pointer,
            second_packed_pointer,
            input_pointer,
            expert_pointer,
            offsets_pointer,
            shape.num_experts,
            shape.num_groups,
            shape.rows,
            shape.out_features,
            shape.in_features,
            shape.bytes_per_expert,
            stream);
    });
    check_hip(hipGetLastError(), "grouped_mmq_pair_grad_input_kernel launch");

    return grad_input;
}

Tensor grouped_mmq_cuda(
        const Tensor & input,
        const Tensor & packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t out_features) {
    const GroupedMMQShape shape = validate_grouped_mmq(
        input,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        out_features);
    const int32_t device_index = input.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(device_index);
    Tensor output = new_grouped_output(input, shape);
    Tensor workspace = new_grouped_workspace(input, shape);

    void * stream_pointer = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_pointer));
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);

    const auto * input_pointer = static_cast<const __hip_bfloat16 *>(input.const_data_ptr());
    const auto * packed_pointer = static_cast<const char *>(packed_weight.const_data_ptr());
    const auto * expert_pointer = static_cast<const int64_t *>(expert_indices.const_data_ptr());
    const auto * offsets_pointer = static_cast<const int32_t *>(expert_offsets.const_data_ptr());
    auto * output_pointer = static_cast<__hip_bfloat16 *>(output.mutable_data_ptr());
    auto * workspace_pointer = static_cast<block_q8_1_mmq *>(workspace.mutable_data_ptr());

    dispatch_quant_type(quant_type, [&](auto type_tag) {
        constexpr ggml_type type = decltype(type_tag)::value;
        launch_grouped_quantize<type>(
            input_pointer,
            workspace_pointer,
            shape.rows,
            shape.in_features,
            stream);
        launch_grouped_projection<type>(
            packed_pointer,
            reinterpret_cast<const int *>(workspace_pointer),
            output_pointer,
            expert_pointer,
            offsets_pointer,
            shape.num_experts,
            shape.num_groups,
            shape.rows,
            shape.in_features,
            shape.out_features,
            shape.bytes_per_expert,
            stream);
    });

    return output;
}

std::tuple<Tensor, Tensor> grouped_mmq_pair_cuda(
        const Tensor & input,
        const Tensor & first_packed_weight,
        const Tensor & second_packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t out_features) {
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
        second_shape.num_experts == shape.num_experts &&
        second_shape.bytes_per_expert == shape.bytes_per_expert,
        "paired grouped MMQ weights must have identical physical geometry");

    const int32_t device_index = input.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(device_index);
    Tensor first_output = new_grouped_output(input, shape);
    Tensor second_output = new_grouped_output(input, shape);
    Tensor workspace = new_grouped_workspace(input, shape);

    void * stream_pointer = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_pointer));
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);

    const auto * input_pointer = static_cast<const __hip_bfloat16 *>(input.const_data_ptr());
    const auto * first_packed_pointer = static_cast<const char *>(first_packed_weight.const_data_ptr());
    const auto * second_packed_pointer = static_cast<const char *>(second_packed_weight.const_data_ptr());
    const auto * expert_pointer = static_cast<const int64_t *>(expert_indices.const_data_ptr());
    const auto * offsets_pointer = static_cast<const int32_t *>(expert_offsets.const_data_ptr());
    auto * first_output_pointer = static_cast<__hip_bfloat16 *>(first_output.mutable_data_ptr());
    auto * second_output_pointer = static_cast<__hip_bfloat16 *>(second_output.mutable_data_ptr());
    auto * workspace_pointer = static_cast<block_q8_1_mmq *>(workspace.mutable_data_ptr());

    dispatch_quant_type(quant_type, [&](auto type_tag) {
        constexpr ggml_type type = decltype(type_tag)::value;
        launch_grouped_quantize<type>(
            input_pointer,
            workspace_pointer,
            shape.rows,
            shape.in_features,
            stream);
        launch_grouped_projection<type>(
            first_packed_pointer,
            reinterpret_cast<const int *>(workspace_pointer),
            first_output_pointer,
            expert_pointer,
            offsets_pointer,
            shape.num_experts,
            shape.num_groups,
            shape.rows,
            shape.in_features,
            shape.out_features,
            shape.bytes_per_expert,
            stream);
        launch_grouped_projection<type>(
            second_packed_pointer,
            reinterpret_cast<const int *>(workspace_pointer),
            second_output_pointer,
            expert_pointer,
            offsets_pointer,
            shape.num_experts,
            shape.num_groups,
            shape.rows,
            shape.in_features,
            shape.out_features,
            shape.bytes_per_expert,
            stream);
    });

    return std::make_tuple(std::move(first_output), std::move(second_output));
}

} // namespace

STABLE_TORCH_LIBRARY(torch_ggml_ops, m) {
    m.def("mmq(Tensor input, Tensor packed_weight, int quant_type, int out_features) -> Tensor");
    m.def(
        "mmq_grad_input(Tensor grad_output, Tensor packed_weight, int quant_type, "
        "int in_features) -> Tensor");
    m.def(
        "grouped_mmq_grad_input(Tensor grad_output, Tensor packed_weight, Tensor expert_indices, "
        "Tensor expert_offsets, int quant_type, int in_features) -> Tensor");
    m.def(
        "grouped_mmq_pair_grad_input(Tensor first_grad_output, Tensor second_grad_output, "
        "Tensor first_packed_weight, Tensor second_packed_weight, Tensor expert_indices, "
        "Tensor expert_offsets, int quant_type, int in_features) -> Tensor");
    m.def(
        "grouped_mmq(Tensor input, Tensor packed_weight, Tensor expert_indices, "
        "Tensor expert_offsets, int quant_type, int out_features) -> Tensor");
    m.def(
        "grouped_mmq_pair(Tensor input, Tensor first_packed_weight, Tensor second_packed_weight, "
        "Tensor expert_indices, Tensor expert_offsets, int quant_type, int out_features) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(torch_ggml_ops, CUDA, m) {
    m.impl("mmq", TORCH_BOX(&mmq_cuda));
    m.impl("mmq_grad_input", TORCH_BOX(&mmq_grad_input_cuda));
    m.impl("grouped_mmq_grad_input", TORCH_BOX(&grouped_mmq_grad_input_cuda));
    m.impl("grouped_mmq_pair_grad_input", TORCH_BOX(&grouped_mmq_pair_grad_input_cuda));
    m.impl("grouped_mmq", TORCH_BOX(&grouped_mmq_cuda));
    m.impl("grouped_mmq_pair", TORCH_BOX(&grouped_mmq_pair_cuda));
}

extern "C" PyObject * PyInit__C(void) {
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "_C",
        nullptr,
        -1,
        nullptr,
    };
    return PyModule_Create(&module);
}
