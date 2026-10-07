#pragma once

#include "generated/mmq_bundle_table.cuh"

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace torch_ggml_ops::mmq_bundle {

void require_exact_deployment(
    int operation,
    std::int32_t quant_type,
    int m,
    int n,
    int k);

// The producer is the activation Q8_1 kernel the resolved record bakes in.
MMQKernelIndex exact_deployment_producer(
    int operation,
    std::int32_t quant_type,
    int m,
    int n,
    int k);

int exact_deployment_row_task_rows(
    int operation,
    std::int32_t quant_type,
    int m,
    int n,
    int k);

int exact_deployment_row_task_capacity(
    int operation,
    std::int32_t quant_type,
    int m,
    int n,
    int k);

// Slice count of one split-contraction dense backward record, or zero when the
// record runs as a single launch.
int exact_deployment_split_slices(
    int operation,
    std::int32_t quant_type,
    int m,
    int n,
    int k);

void launch_quantize(
    MMQKernelIndex producer,
    const void * input,
    void * output,
    std::int64_t rows,
    std::int64_t rows_padded,
    std::int64_t in_features,
    hipStream_t stream);

void launch_dense_forward(
    std::int32_t quant_type,
    const char * packed,
    const int * activations,
    void * output,
    int rows,
    int rows_padded,
    int in_features,
    int out_features,
    hipStream_t stream);

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
    hipStream_t stream);

void launch_fixed_grouped_forward(
    const char * packed,
    const int * activations,
    void * output,
    int tokens,
    int in_features,
    int out_features,
    std::int64_t bytes_per_group,
    hipStream_t stream);

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
    hipStream_t stream);

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
    hipStream_t stream);

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
    hipStream_t stream);

// `weight_block_values` is the quantization block width of `quant_type`. A HIP
// backward record strides its packed row in those blocks.
void launch_dense_backward(
    std::int32_t quant_type,
    const void * grad_output,
    const char * packed_weight,
    void * grad_input,
    int rows,
    int out_features,
    int in_features,
    int weight_block_values,
    hipStream_t stream);

void launch_fixed_grouped_backward(
    const void * grad_output,
    const char * packed_weight,
    void * grad_input,
    int tokens,
    int in_features,
    int out_features,
    std::int64_t bytes_per_group,
    hipStream_t stream);

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
    hipStream_t stream);

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
    hipStream_t stream);

// Device row-task variants of the routed backward launches. The task bank is
// built by the shared setup artifact from the route descriptors before the
// body runs, so the caller owns both the descriptors and the bank.
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
    hipStream_t stream);

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
    hipStream_t stream);

// Split-contraction dense backward. The slice body writes FP32 partials across
// the record's slice grid and the reduction sums them into the BF16 gradient.
void launch_dense_backward_split(
    std::int32_t quant_type,
    const void * grad_output,
    const char * packed_weight,
    void * partials,
    int rows,
    int out_features,
    int in_features,
    int weight_block_values,
    hipStream_t stream);

void launch_dense_backward_split_reduce(
    const void * partials,
    void * grad_input,
    int rows,
    int in_features,
    int slices,
    hipStream_t stream);

} // namespace torch_ggml_ops::mmq_bundle
