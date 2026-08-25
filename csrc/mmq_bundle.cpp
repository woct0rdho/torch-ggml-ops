#include "mmq_bundle.h"
#include "mmq_bundle_loader.h"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <limits>
#include <string>

namespace torch_ggml_ops::mmq_bundle {
namespace {

static_assert(kMMQKernelSymbols.size() <=
              std::numeric_limits<MMQKernelIndex>::max());

using detail::fail;

const MMQDeploymentRecord & exact_record(
        int operation,
        std::int32_t quant_type,
        int m,
        int n,
        int k) {
    for (const MMQDeploymentRecord & record : kMMQDeployments) {
        if (record.operation == operation && record.quant_type == quant_type &&
            record.m == m && record.n == n && record.k == k) {
            return record;
        }
    }
    fail(
        "unsupported exact deployment key operation=" +
        std::to_string(operation) + " quant_type=" +
        std::to_string(quant_type) + " M=" + std::to_string(m) +
        " N=" + std::to_string(n) + " K=" + std::to_string(k));
}

void launch_record(
        const MMQDeploymentRecord & record,
        unsigned int grid_y,
        hipStream_t stream,
        void ** arguments) {
    detail::launch_kernel(
        record.kernel,
        record.grid_x,
        grid_y,
        record.grid_z,
        record.block_x,
        record.block_y,
        record.block_z,
        stream,
        arguments);
}

MMQKernelIndex quantize_kernel(std::int32_t quant_type) {
    if (quant_type == kQuantQ2_K) {
        return kQuantizeQ81F16D2S6;
    }
    if (quant_type == kQuantQ4_K || quant_type == kQuantQ5_K) {
        return kQuantizeQ81F16D4S4;
    }
    return kQuantizeQ81F32D4;
}

} // namespace

void require_exact_deployment(
        int operation,
        std::int32_t quant_type,
        int m,
        int n,
        int k) {
    (void)exact_record(operation, quant_type, m, n, k);
}

int exact_deployment_row_task_rows(
        int operation,
        std::int32_t quant_type,
        int m,
        int n,
        int k) {
    return exact_record(operation, quant_type, m, n, k).row_task_rows;
}

int exact_deployment_row_task_capacity(
        int operation,
        std::int32_t quant_type,
        int m,
        int n,
        int k) {
    return exact_record(operation, quant_type, m, n, k).row_task_capacity;
}

void launch_quantize(
        std::int32_t quant_type,
        const void * input,
        void * output,
        std::int64_t rows,
        std::int64_t rows_padded,
        std::int64_t in_features,
        hipStream_t stream) {
    void * arguments[]{&input, &output, &rows, &rows_padded, &in_features};
    detail::launch_kernel(
        quantize_kernel(quant_type),
        static_cast<unsigned int>(rows),
        1,
        1,
        512,
        1,
        1,
        stream,
        arguments);
}

void launch_dense_forward(
        std::int32_t quant_type,
        const char * packed,
        const int * activations,
        void * output,
        int rows,
        int rows_padded,
        int in_features,
        int out_features,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kOrdinaryForward, quant_type, rows, out_features, in_features);
    unsigned int nrows_weight = static_cast<unsigned int>(out_features);
    unsigned int nrows_activation = static_cast<unsigned int>(rows);
    unsigned int nrows_activation_padded = static_cast<unsigned int>(rows_padded);
    unsigned int blocks_per_weight_row =
        static_cast<unsigned int>(in_features / 256);
    void * arguments[]{
        &packed, &activations, &output, &nrows_weight, &nrows_activation,
        &nrows_activation_padded, &blocks_per_weight_row,
    };
    launch_record(record, record.grid_y, stream, arguments);
}

void launch_grouped_row_task_setup(
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        std::int32_t * task_count,
        std::int32_t * task_experts,
        std::int32_t * task_row_starts,
        std::int32_t * task_row_ends,
        int num_experts,
        int num_groups,
        int rows,
        int row_tile,
        hipStream_t stream) {
    void * arguments[]{
        &expert_indices, &expert_offsets, &task_count, &task_experts,
        &task_row_starts, &task_row_ends, &num_experts, &num_groups,
        &rows, &row_tile,
    };
    detail::launch_kernel(
        kGroupedRowTaskSetup,
        1, 1, 1,
        256, 1, 1,
        stream,
        arguments);
}

void launch_fixed_grouped_forward(
        const char * packed,
        const int * activations,
        void * output,
        int tokens,
        int in_features,
        int out_features,
        std::int64_t bytes_per_group,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kFixedGroupedForward, kQuantQ8_0, tokens, out_features, in_features);
    unsigned int tokens_value = static_cast<unsigned int>(tokens);
    unsigned int out_features_value = static_cast<unsigned int>(out_features);
    std::uint64_t bytes_value = static_cast<std::uint64_t>(bytes_per_group);
    void * arguments[]{
        &packed, &activations, &output, &tokens_value,
        &out_features_value, &bytes_value,
    };
    launch_record(record, record.grid_y, stream, arguments);
}

void launch_grouped_forward(
        std::int32_t quant_type,
        const char * packed,
        const int * activations,
        void * output,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int in_features,
        int out_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedForward, quant_type, rows, out_features, in_features);
    unsigned int num_experts_value = static_cast<unsigned int>(num_experts);
    unsigned int nrows_weight = static_cast<unsigned int>(out_features);
    unsigned int nrows_activation = static_cast<unsigned int>(rows);
    unsigned int blocks_per_weight_row =
        static_cast<unsigned int>(in_features / 256);
    std::uint64_t bytes_value = static_cast<std::uint64_t>(bytes_per_expert);
    void * arguments[]{
        &packed, &activations, &output, &expert_indices, &expert_offsets,
        &num_experts_value, &nrows_weight, &nrows_activation,
        &blocks_per_weight_row, &bytes_value,
    };
    launch_record(
        record, static_cast<unsigned int>(num_groups), stream, arguments);
}

void launch_grouped_pair_forward_row_tasks(
        std::int32_t quant_type,
        const char * first_packed,
        const char * second_packed,
        const int * activations,
        void * first_output,
        void * second_output,
        const std::int32_t * task_count,
        const std::int32_t * task_experts,
        const std::int32_t * task_row_starts,
        const std::int32_t * task_row_ends,
        int max_tasks,
        int num_experts,
        int rows,
        int in_features,
        int out_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedForwardPair, quant_type, rows, out_features, in_features);
    if (record.ownership != 1) {
        fail("exact paired-forward deployment does not use row tasks");
    }
    unsigned int num_experts_value = static_cast<unsigned int>(num_experts);
    unsigned int nrows_weight = static_cast<unsigned int>(out_features);
    unsigned int nrows_activation = static_cast<unsigned int>(rows);
    unsigned int blocks_per_weight_row =
        static_cast<unsigned int>(in_features / 256);
    std::uint64_t bytes_value = static_cast<std::uint64_t>(bytes_per_expert);
    void * arguments[]{
        &first_packed, &second_packed, &activations,
        &first_output, &second_output, &task_count, &task_experts,
        &task_row_starts, &task_row_ends, &num_experts_value,
        &nrows_weight, &nrows_activation, &blocks_per_weight_row, &bytes_value,
    };
    launch_record(
        record, static_cast<unsigned int>(max_tasks), stream, arguments);
}

void launch_grouped_pair_forward(
        std::int32_t quant_type,
        const char * first_packed,
        const char * second_packed,
        const int * activations,
        void * first_output,
        void * second_output,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int in_features,
        int out_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedForwardPair, quant_type, rows, out_features, in_features);
    if (record.ownership != 0) {
        fail("exact paired-forward deployment requires row tasks");
    }
    unsigned int num_experts_value = static_cast<unsigned int>(num_experts);
    unsigned int nrows_weight = static_cast<unsigned int>(out_features);
    unsigned int nrows_activation = static_cast<unsigned int>(rows);
    unsigned int blocks_per_weight_row =
        static_cast<unsigned int>(in_features / 256);
    std::uint64_t bytes_value = static_cast<std::uint64_t>(bytes_per_expert);
    void * arguments[]{
        &first_packed, &second_packed, &activations,
        &first_output, &second_output, &expert_indices, &expert_offsets,
        &num_experts_value, &nrows_weight, &nrows_activation,
        &blocks_per_weight_row, &bytes_value,
    };
    launch_record(
        record, static_cast<unsigned int>(num_groups), stream, arguments);
}

void launch_dense_backward(
        std::int32_t quant_type,
        const void * grad_output,
        const char * packed_weight,
        void * grad_input,
        int rows,
        int out_features,
        int in_features,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kOrdinaryBackward, quant_type, rows, in_features, out_features);
    unsigned int rows_value = static_cast<unsigned int>(rows);
    unsigned int out_features_value = static_cast<unsigned int>(out_features);
    unsigned int in_features_value = static_cast<unsigned int>(in_features);
    unsigned int blocks_per_weight_row =
        static_cast<unsigned int>(in_features / 256);
    void * arguments[]{
        &grad_output, &packed_weight, &grad_input, &rows_value,
        &out_features_value, &in_features_value, &blocks_per_weight_row,
    };
    launch_record(record, record.grid_y, stream, arguments);
}

void launch_fixed_grouped_backward(
        const void * grad_output,
        const char * packed_weight,
        void * grad_input,
        int tokens,
        int in_features,
        int out_features,
        std::int64_t bytes_per_group,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kFixedGroupedBackward, kQuantQ8_0, tokens, in_features, out_features);
    unsigned int tokens_value = static_cast<unsigned int>(tokens);
    unsigned int out_features_value = static_cast<unsigned int>(out_features);
    std::uint64_t bytes_value = static_cast<std::uint64_t>(bytes_per_group);
    void * arguments[]{
        &grad_output, &packed_weight, &grad_input, &tokens_value,
        &out_features_value, &bytes_value,
    };
    launch_record(record, record.grid_y, stream, arguments);
}

void launch_grouped_backward(
        std::int32_t quant_type,
        const void * grad_output,
        const char * packed_weight,
        void * grad_input,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int out_features,
        int in_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedBackward, quant_type, rows, in_features, out_features);
    std::int64_t bytes_value = bytes_per_expert;
    void * arguments[]{
        &grad_output, &packed_weight, &grad_input,
        &expert_indices, &expert_offsets, &num_experts, &rows, &bytes_value,
    };
    launch_record(
        record, static_cast<unsigned int>(num_groups), stream, arguments);
}

void launch_grouped_pair_backward(
        std::int32_t quant_type,
        const void * first_grad_output,
        const void * second_grad_output,
        const char * first_packed_weight,
        const char * second_packed_weight,
        void * grad_input,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int out_features,
        int in_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedBackwardPair, quant_type, rows, in_features, out_features);
    std::int64_t bytes_value = bytes_per_expert;
    void * arguments[]{
        &first_grad_output, &second_grad_output,
        &first_packed_weight, &second_packed_weight, &grad_input,
        &expert_indices, &expert_offsets, &num_experts, &rows, &bytes_value,
    };
    launch_record(
        record,
        static_cast<unsigned int>(num_groups * record.route_split_factor),
        stream,
        arguments);
}

} // namespace torch_ggml_ops::mmq_bundle
