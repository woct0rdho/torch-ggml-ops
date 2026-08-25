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

void launch_quantize(
    std::int32_t quant_type,
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

void launch_dense_backward(
    std::int32_t quant_type,
    const void * grad_output,
    const char * packed_weight,
    void * grad_input,
    int rows,
    int out_features,
    int in_features,
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

} // namespace torch_ggml_ops::mmq_bundle
