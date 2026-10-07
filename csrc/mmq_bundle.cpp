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
        // A GGTensile artifact owns its LDS in the code object's group segment
        // and records zero here. A HIP control records its request.
        record.dynamic_shared_bytes,
        stream,
        arguments);
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

MMQKernelIndex exact_deployment_producer(
        int operation,
        std::int32_t quant_type,
        int m,
        int n,
        int k) {
    return exact_record(operation, quant_type, m, n, k).producer;
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

int exact_deployment_split_slices(
        int operation,
        std::int32_t quant_type,
        int m,
        int n,
        int k) {
    return exact_record(operation, quant_type, m, n, k).split_slices;
}

void launch_quantize(
        MMQKernelIndex producer,
        const void * input,
        void * output,
        std::int64_t rows,
        std::int64_t rows_padded,
        std::int64_t in_features,
        hipStream_t stream) {
    void * arguments[]{&input, &output, &rows, &rows_padded, &in_features};
    detail::launch_kernel(
        producer,
        static_cast<unsigned int>(rows),
        1,
        1,
        512,
        1,
        1,
        0,
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
        int out_features,
        int in_features,
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
        0,
        stream,
        arguments);
}

void launch_fixed_grouped_forward(
        const char * packed,
        const int * activations,
        void * output,
        int tokens,
        int out_features,
        int in_features,
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

void launch_grouped_forward_body(
        const MMQDeploymentRecord & record,
        const char * packed,
        const int * activations,
        void * output,
        const std::int64_t * expert_indices,
        const std::int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int out_features,
        int in_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
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
        int out_features,
        int in_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedForward, quant_type, rows, out_features, in_features);
    launch_grouped_forward_body(
        record,
        packed,
        activations,
        output,
        expert_indices,
        expert_offsets,
        num_experts,
        num_groups,
        rows,
        out_features,
        in_features,
        bytes_per_expert,
        stream);
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
        int out_features,
        int in_features,
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
        int out_features,
        int in_features,
        std::int64_t bytes_per_expert,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedForwardPair, quant_type, rows, out_features, in_features);
    if (record.ownership == 2) {
        // The deployed family has no fused pair body: the pair operation runs
        // the routed single-projection body once per packed bank over the same
        // route bank and the same activation workspace.
        for (const auto & pair : {std::make_pair(first_packed, first_output),
                                  std::make_pair(second_packed, second_output)}) {
            launch_grouped_forward_body(
                record,
                pair.first,
                activations,
                pair.second,
                expert_indices,
                expert_offsets,
                num_experts,
                num_groups,
                rows,
                out_features,
                in_features,
                bytes_per_expert,
                stream);
        }
        return;
    }
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
        int weight_block_values,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kOrdinaryBackward, quant_type, rows, out_features, in_features);
    unsigned int rows_value = static_cast<unsigned int>(rows);
    unsigned int out_features_value = static_cast<unsigned int>(out_features);
    unsigned int in_features_value = static_cast<unsigned int>(in_features);
    // A GGTensile backward artifact walks the contraction in 256-value stages.
    // A HIP control strides the packed row in its own quantization blocks.
    unsigned int blocks_per_weight_row =
        record.implementation == kImplementationHip
        ? static_cast<unsigned int>(in_features / weight_block_values)
        : static_cast<unsigned int>(in_features / 256);
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
        int out_features,
        int in_features,
        std::int64_t bytes_per_group,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kFixedGroupedBackward, kQuantQ8_0, tokens, out_features, in_features);
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
        kGroupedBackward, quant_type, rows, out_features, in_features);
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
        kGroupedBackwardPair, quant_type, rows, out_features, in_features);
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

void launch_grouped_backward_row_tasks(
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
        void * task_count,
        void * task_experts,
        void * task_row_starts,
        void * task_row_ends,
        int task_capacity,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedBackward, quant_type, rows, out_features, in_features);
    if (record.ownership != 1 || record.row_task_rows == 0) {
        fail("exact grouped-backward deployment does not use row tasks");
    }
    std::int64_t bytes_value = bytes_per_expert;
    launch_grouped_row_task_setup(
        expert_indices,
        expert_offsets,
        static_cast<std::int32_t *>(task_count),
        static_cast<std::int32_t *>(task_experts),
        static_cast<std::int32_t *>(task_row_starts),
        static_cast<std::int32_t *>(task_row_ends),
        num_experts,
        num_groups,
        rows,
        record.row_task_rows,
        stream);
    void * arguments[]{
        &grad_output, &packed_weight, &grad_input, &task_count, &task_experts,
        &task_row_starts, &task_row_ends, &bytes_value,
    };
    launch_record(
        record, static_cast<unsigned int>(task_capacity), stream, arguments);
}

void launch_grouped_pair_backward_row_tasks(
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
        void * task_count,
        void * task_experts,
        void * task_row_starts,
        void * task_row_ends,
        int task_capacity,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kGroupedBackwardPair, quant_type, rows, out_features, in_features);
    if (record.ownership != 1 || record.row_task_rows == 0) {
        fail("exact paired-backward deployment does not use row tasks");
    }
    std::int64_t bytes_value = bytes_per_expert;
    launch_grouped_row_task_setup(
        expert_indices,
        expert_offsets,
        static_cast<std::int32_t *>(task_count),
        static_cast<std::int32_t *>(task_experts),
        static_cast<std::int32_t *>(task_row_starts),
        static_cast<std::int32_t *>(task_row_ends),
        num_experts,
        num_groups,
        rows,
        record.row_task_rows,
        stream);
    void * arguments[]{
        &first_grad_output, &second_grad_output,
        &first_packed_weight, &second_packed_weight, &grad_input,
        &task_count, &task_experts, &task_row_starts, &task_row_ends,
        &num_experts, &rows, &bytes_value,
    };
    launch_record(
        record, static_cast<unsigned int>(task_capacity), stream, arguments);
}

void launch_dense_backward_split(
        std::int32_t quant_type,
        const void * grad_output,
        const char * packed_weight,
        void * partials,
        int rows,
        int out_features,
        int in_features,
        int weight_block_values,
        hipStream_t stream) {
    const MMQDeploymentRecord & record = exact_record(
        kOrdinaryBackward, quant_type, rows, out_features, in_features);
    if (record.split_slices <= 0) {
        fail("exact dense-backward deployment is not a split-contraction control");
    }
    unsigned int rows_value = static_cast<unsigned int>(rows);
    unsigned int out_features_value = static_cast<unsigned int>(out_features);
    unsigned int in_features_value = static_cast<unsigned int>(in_features);
    unsigned int blocks_per_weight_row =
        record.implementation == kImplementationHip
        ? static_cast<unsigned int>(in_features / weight_block_values)
        : static_cast<unsigned int>(in_features / 256);
    unsigned int split_chunk = static_cast<unsigned int>(record.split_chunk);
    void * arguments[]{
        &grad_output, &packed_weight, &partials, &rows_value,
        &out_features_value, &in_features_value, &blocks_per_weight_row,
        &split_chunk,
    };
    launch_record(record, record.grid_y, stream, arguments);
}

void launch_dense_backward_split_reduce(
        const void * partials,
        void * grad_input,
        int rows,
        int in_features,
        int slices,
        hipStream_t stream) {
    const std::int64_t count = static_cast<std::int64_t>(rows) * in_features;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::int64_t>(1024, std::max<std::int64_t>(1, (count + 255) / 256)));
    void * arguments[]{
        &partials, &grad_input, &rows, &in_features, &slices,
    };
    detail::launch_kernel(
        kDenseBackwardSplitKReduce,
        blocks,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        arguments);
}

} // namespace torch_ggml_ops::mmq_bundle
